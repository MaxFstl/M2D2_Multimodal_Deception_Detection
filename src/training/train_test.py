import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn

from src.data.loaders import create_dataloaders
from src.models.audio_model import W2V2_Model
from src.models.fusion_model import Fusion
from src.utils.load_config import load_config

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
)

from src.models.visual_model import ViT_model
from src.models.text_model import TextModel


def parse_options():
    parser = argparse.ArgumentParser(description="WILTY baseline repo")
    ##### TRAINING DYNAMICS
    config = load_config()
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="the gpu id used for predict"
    )
    parser.add_argument(
        "--device_ID", type=str, default="cuda:0", help="the gpu id used for predict"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="initial learning rate"
    )  # default=0.001
    parser.add_argument(
        "--when", type=int, default=15, help="when to decay learning rate (default: 20)"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="initial batchsize")
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="total training epochs"
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument(
        "--multi", action="store_true", help="multitask learning with multiple losses"
    )
    ##### UNIMODAL ARCHITECTURE

    parser.add_argument(
        "--num_encoders",
        type=int,
        default=4,
        help="number of transformer encoders for each modality",
    )
    parser.add_argument(
        "--adapter", action="store_true", help="indicator of using adapter"
    )
    parser.add_argument(
        "--adapter_type",
        type=str,
        default="efficient_conv",
        help="indicate which type of adapter to use - full_conv or efficient_conv or nlp",
    )
    ##### DATA

    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument(
        "--audio_path",
        type=str,
        default=config["paths"]["audio_dir"],
        help="audio dataset root dir",
    )
    parser.add_argument(
        "--annos_path",
        type=str,
        default=config["paths"]["annotations"],
        help="audio dataset root dir",
    )
    parser.add_argument(
        "--visual_path",
        type=str,
        default=config["paths"]["image_dir"],
        help="visual (FACE) dataset root dir",
    )
    parser.add_argument(
        "--text_path",
        type=str,
        default=config["paths"]["text_dir"],
        help="text transcripts root dir",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=config["paths"]["annotations"],
        help="annotations file path",
    )
    parser.add_argument(
        "--text_max_length",
        type=int,
        default=512,
        help="max length for text tokenization",
    )
    parser.add_argument(
        "--use_mock_text",
        action="store_true",
        help="use mock text instead of real transcripts",
    )

    #### TRAINING LOGS

    parser.add_argument(
        "--log", type=str, default="logs", help="log and save model name"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="directory to save model checkpoints",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--save_best_only",
        action="store_true",
        default=True,
        help="only save checkpoint when validation accuracy improves",
    )
    parser.add_argument(
        "--protocols",
        type=list,
        default=[["train_fold3.csv", "test_fold3.csv"]],
        help="protocols for train/test",
    )

    """
    parser.add_argument('--protocols', type=list,
                        default=[['train_file_1.csv', 'test_file_1.csv'], ['train_file_2.csv', 'test_file_2.csv'],
                                 ['train_file_3.csv', 'test_file_3.csv'], ['train_file_4.csv', 'test_file_4.csv'],
                                 ['train_file_5.csv', 'test_file_5.csv']],
                        help='protocols for train/test')                                 
    ['train_fold1.csv', 'test_fold1.csv'], ['train_fold2.csv', 'test_fold2.csv'],
                                 ['train_fold3.csv', 'test_fold3.csv'], ['long.csv', 'short.csv'],
                                 ['short.csv', 'long.csv'],
                                 ['male.csv', 'female.csv'], ['female.csv', 'male.csv']
                                 
    ['WILTY.csv', 'bag_of_lies.csv'], ['WILTY.csv', 'RLT_annotations.csv'], 
    ['bol_audio_visual.csv', 'bag_of_lies.csv'], ['bol_audio_visual.csv', 'RLT_annotations.csv']
    """
    parser.add_argument(
        "--full_annotation", type=str, default="/DOLOS/data/protocols/WILTY_binary.csv"
    )

    parser.add_argument("--model_name", type=str, default="DOLOS_")
    parser.add_argument(
        "--model_to_train",
        type=str,
        default="fusion",
        help="specify vision or audio or fusion- indiciates the model to be trained",
    )
    parser.add_argument(
        "--fusion_type", type=str, default="cross2", help="modality fusion type"
    )

    opts = parser.parse_args()
    torch.manual_seed(opts.seed)
    opts.device = torch.device(opts.device)
    # if "fusion" in opts.model_to_train:
    #     opts.model_to_train = opts.model_to_train + "_" + opts.fusion_type

    if opts.adapter:
        opts.model_name = (
            opts.model_name
            + opts.model_to_train
            + "_Encoders_"
            + str(opts.num_encoders)
            + "_Adapter_"
            + str(opts.adapter)
            + "_type_"
            + str(opts.adapter_type)
        )
    else:
        opts.model_name = (
            opts.model_name
            + opts.model_to_train
            + "_Encoders_"
            + str(opts.num_encoders)
            + "_Adapter_"
            + str(opts.adapter)
        )
    if not os.path.exists(opts.log):
        os.makedirs(opts.log)
    
    if not os.path.exists(opts.checkpoint_dir):
        os.makedirs(opts.checkpoint_dir)

    return opts


def save_checkpoint(
    checkpoint_dir: str,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    config: dict,
    is_best: bool = False,
) -> str:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "config": config,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
        torch.save(checkpoint, best_path)
        return best_path
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None,
) -> tuple:
    if device is None:
        device = torch.device("cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return (
        checkpoint["epoch"],
        checkpoint["best_acc"],
        checkpoint.get("config", {}),
    )


def train_one_epoch(
    args,
    train_data_loader,
    model,
    optimizer,
    loss_fn,
    loss_audio,
    loss_vision,
    loss_text,
):
    epoch_loss = []
    epoch_predictions = []
    epoch_labels = []
    start_time = time.time()

    model.train()

    if args.model_to_train == "audio":
        for i, batch in enumerate(train_data_loader):
            waves = batch["audio"].squeeze(1)
            waves = waves.to(args.device)
            labels = batch["label"].to(args.device)

            optimizer.zero_grad()
            preds = model(waves)
            _loss = loss_fn(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            _loss.backward()
            optimizer.step()

            epoch_predictions.append(torch.argmax(preds, dim=1))
            epoch_labels.append(labels)

            if i % 10 == 0:
                print("iter {}, loss {:.5f}".format(str(i), loss))

    elif args.model_to_train == "vision":
        for i, batch in enumerate(train_data_loader):
            faces = batch["video"].to(args.device)
            labels = batch["label"].to(args.device)

            optimizer.zero_grad()
            preds = model(faces)
            _loss = loss_fn(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            _loss.backward()
            optimizer.step()

            epoch_predictions.append(torch.argmax(preds, dim=1))
            epoch_labels.append(labels)

            if i % 10 == 0:
                print("iter {}, loss {:.5f}".format(str(i), loss))

    elif args.model_to_train == "text":
        for i, batch in enumerate(train_data_loader):
            input_ids = batch["text_ids"].to(args.device)
            attention_mask = batch["text_mask"].to(args.device)
            labels = batch["label"].to(args.device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            _loss = loss_fn(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            _loss.backward()
            optimizer.step()

            epoch_predictions.append(torch.argmax(preds, dim=1))
            epoch_labels.append(labels)

            if i % 10 == 0:
                print("iter {}, loss {:.5f}".format(str(i), loss))

    else:  # fusion models
        for i, batch in enumerate(train_data_loader):
            waves = batch["audio"].squeeze(1).to(args.device)
            faces = batch["video"].to(args.device)
            input_ids = batch["text_ids"].to(args.device)
            attention_mask = batch["text_mask"].to(args.device)
            labels = batch["label"].to(args.device)

            optimizer.zero_grad()

            preds, a_preds, v_preds, t_preds = model(
                waves, faces, input_ids, attention_mask
            )

            _loss = loss_fn(preds, labels)

            if args.multi:
                _loss_a = loss_audio(a_preds, labels)
                _loss_v = loss_vision(v_preds, labels)
                _loss_t = loss_text(t_preds, labels)
                _loss = _loss + _loss_a + _loss_v + _loss_t
            loss = _loss.item()
            epoch_loss.append(loss)

            _loss.backward()
            optimizer.step()

            epoch_predictions.append(torch.argmax(preds, dim=1))
            epoch_labels.append(labels)

            if i % 10 == 0:
                print("iter {}, loss {:.5f}".format(str(i), loss))

    epoch_predictions = torch.cat(epoch_predictions)
    epoch_labels = torch.cat(epoch_labels)

    end_time = time.time()
    _ = end_time - start_time

    epoch_loss = np.mean(epoch_loss)

    return epoch_loss, epoch_predictions, epoch_labels, None, None


def val_one_epoch(
    args, val_data_loader, model, loss_fn, loss_audio, loss_vision, loss_text
):
    epoch_loss = []
    epoch_predictions = []
    epoch_labels = []
    start_time = time.time()

    model.eval()

    with torch.no_grad():
        if args.model_to_train == "audio":
            for batch in val_data_loader:
                waves = batch["audio"].squeeze(1).to(args.device)
                labels = batch["label"].to(args.device)

                preds = model(waves)
                _loss = loss_fn(preds, labels)
                loss = _loss.item()
                epoch_loss.append(loss)

                epoch_predictions.append(torch.argmax(preds, dim=1))
                epoch_labels.append(labels)

        elif args.model_to_train == "vision":
            for batch in val_data_loader:
                faces = batch["video"].to(args.device)
                labels = batch["label"].to(args.device)

                preds = model(faces)
                _loss = loss_fn(preds, labels)
                loss = _loss.item()
                epoch_loss.append(loss)

                epoch_predictions.append(torch.argmax(preds, dim=1))
                epoch_labels.append(labels)

        elif args.model_to_train == "text":
            for batch in val_data_loader:
                input_ids = batch["text_ids"].to(args.device)
                attention_mask = batch["text_mask"].to(args.device)
                labels = batch["label"].to(args.device)

                preds = model(input_ids, attention_mask)
                _loss = loss_fn(preds, labels)
                loss = _loss.item()
                epoch_loss.append(loss)

                epoch_predictions.append(torch.argmax(preds, dim=1))
                epoch_labels.append(labels)

        else:  # fusion
            for batch in val_data_loader:
                waves = batch["audio"].squeeze(1).to(args.device)
                faces = batch["video"].to(args.device)
                input_ids = batch["text_ids"].to(args.device)
                attention_mask = batch["text_mask"].to(args.device)
                labels = batch["label"].to(args.device)

                preds, a_preds, v_preds, t_preds = model(
                    waves, faces, input_ids, attention_mask
                )

                _loss = loss_fn(preds, labels)
                if args.multi:
                    _loss_a = loss_audio(a_preds, labels)
                    _loss_v = loss_vision(v_preds, labels)
                    _loss_t = loss_text(t_preds, labels)
                    _loss = _loss + _loss_a + _loss_v + _loss_t
                loss = _loss.item()
                epoch_loss.append(loss)

                epoch_predictions.append(torch.argmax(preds, dim=1))
                epoch_labels.append(labels)

    epoch_predictions = torch.cat(epoch_predictions)
    epoch_labels = torch.cat(epoch_labels)

    end_time = time.time()
    _ = end_time - start_time

    epoch_loss = np.mean(epoch_loss)
    return epoch_loss, epoch_predictions, epoch_labels, None, None


def evaluation(labels, preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    auc_score = auc(fpr, tpr)
    return acc, f1, auc_score


def evaluation_multi(labels, preds, labels2, preds2):
    acc, f1, auc_score = evaluation(labels, preds)

    acc2, f1_2, auc_2 = {}, {}, {}
    for i in range(labels2.shape[1]):
        acc2[i] = accuracy_score(labels2[:, i], preds2[:, i])
        f1_2[i] = f1_score(labels2[:, i], preds2[:, i])
        fpr, tpr, thresholds = roc_curve(labels2[:, i], preds2[:, i], pos_label=1)
        auc_2[i] = auc(fpr, tpr)
    return acc, f1, auc_score, acc2, f1_2, auc_2


def load_text_transcripts(text_dir: str) -> dict:
    texts = {}
    if os.path.exists(text_dir):
        for fname in os.listdir(text_dir):
            if fname.endswith(".txt"):
                clip_name = fname.replace(".txt", "")
                with open(os.path.join(text_dir, fname), "r", encoding="utf-8") as f:
                    texts[clip_name] = f.read().strip()
    return texts


def train_test(log_name, args):
    f = open(log_name, "a")

    config = load_config()

    f.write(
        "\n\nUsing stratified train/val/test split ({}/{}/{})".format(
            config["data"]["train_split"],
            config["data"]["val_split"],
            config["data"]["test_split"],
        )
    )

    if args.model_to_train in ["text", "fusion"]:
        texts_dict = load_text_transcripts(args.text_path)
    else:
        texts_dict = None

    if args.model_to_train in ["audio", "vision", "text", "fusion"]:
        loader_config = {
            "batch_size": args.batch_size,
            "num_frames": config["data"]["video"]["num_frames"],
            "frame_size": config["data"]["video"]["resize"][0],
            "num_audio_tokens": 64,
            "train_split": config["data"]["train_split"],
            "val_split": config["data"]["val_split"],
            "test_split": config["data"]["test_split"],
            "class_weights": config["training"]["class_weights"],
        }
        train_loader, val_loader, test_loader = create_dataloaders(
            annotations_file=args.annos_path,
            audio_dir=args.audio_path,
            img_dir=args.visual_path,
            config=loader_config,
            texts=texts_dict,
            use_mock_text=args.use_mock_text,
            num_workers=4,
        )

    else:
        raise Exception(
            " please indicate which model to train: audio, vision, text or fusion ..."
        )

    print("\t Dataset Loaded")

    if args.model_to_train == "audio":
        model = W2V2_Model(args.num_encoders, args.adapter, args.adapter_type)
        model.to(args.device)
    elif args.model_to_train == "vision":
        model = ViT_model(args.num_encoders, args.adapter, args.adapter_type)
        model.to(args.device)
    elif args.model_to_train == "text":
        model = TextModel(args.num_encoders, args.adapter, args.adapter_type)
        model.to(args.device)
    else:
        model = Fusion(
            args.fusion_type,
            args.num_encoders,
            args.adapter,
            args.adapter_type,
            args.multi,
            num_frames=config["data"]["video"]["num_frames"],
        )
        model.to(args.device)
    print("\t Model Loaded")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_fn = nn.CrossEntropyLoss()
    if args.multi:
        loss_audio = nn.CrossEntropyLoss()
        loss_vision = nn.CrossEntropyLoss()
        loss_text = nn.CrossEntropyLoss()
    else:
        loss_audio = loss_vision = loss_text = None

    best_acc = 0.0
    start_epoch = 0
    results = ""
    val_results = ""
    test_results = ""
    multi_task_result = ""

    model_config = {
        "model_to_train": args.model_to_train,
        "num_encoders": args.num_encoders,
        "adapter": args.adapter,
        "adapter_type": args.adapter_type,
        "fusion_type": getattr(args, "fusion_type", None),
        "multi": args.multi,
        "num_frames": config["data"]["video"]["num_frames"],
    }

    if args.load_checkpoint:
        print(f"\t Loading checkpoint from {args.load_checkpoint}")
        start_epoch, best_acc, _ = load_checkpoint(
            args.load_checkpoint, model, optimizer, args.device
        )
        start_epoch += 1
        print(f"\t Resuming from epoch {start_epoch}, best_acc: {best_acc:.5f}")

    print("\t Started Training")
    if args.multi:
        print("\t Multitask learning")
    for epoch in range(start_epoch, args.num_epochs):

        if (epoch + 1) % 5 == 0:
            print("\t\t Epoch....", epoch + 1)

        loss, preds, labels, preds_multi, labels_multi = train_one_epoch(
            args,
            train_loader,
            model,
            optimizer,
            loss_fn,
            loss_audio,
            loss_vision,
            loss_text,
        )

        train_acc, train_f1, train_auc = evaluation(
            labels.detach().cpu().numpy(), preds.detach().cpu().numpy()
        )

        val_loss, val_preds, val_labels, val_preds_multi, val_labels_multi = (
            val_one_epoch(
                args,
                val_loader,
                model,
                loss_fn,
                loss_audio,
                loss_vision,
                loss_text,
            )
        )

        val_acc, val_f1, val_auc = evaluation(
            val_labels.detach().cpu().numpy(), val_preds.detach().cpu().numpy()
        )

        print(
            "epoch {}, train_acc {:.5f}, train_f1: {:.5f}, train_auc:{:.5f} "
            "val_acc {:.5f}, val_f1: {:.5f}, val_auc:{:.5f}".format(
                epoch, train_acc, train_f1, train_auc, val_acc, val_f1, val_auc
            )
        )

        f.write(
            "epoch {}, val_acc {:.5f}, val_f1: {:.5f}, val_auc:{:.5f}\n".format(
                epoch, val_acc, val_f1, val_auc
            )
        )

        if val_acc > best_acc:
            best_acc = val_acc
            results = "best val results are acc {:.5f}, f1: {:.5f}, auc:{:.5f} ".format(
                val_acc, val_f1, val_auc
            )
            val_results = classification_report(
                val_labels.cpu().numpy(),
                val_preds.cpu().numpy(),
                target_names=["truth", "deception"],
            )
            
            if args.save_best_only:
                best_path = save_checkpoint(
                    args.checkpoint_dir,
                    args.model_name,
                    model,
                    optimizer,
                    epoch,
                    best_acc,
                    model_config,
                    is_best=True,
                )
                print(f"\t Saved best model to {best_path} (val_acc: {val_acc:.5f})")

    print("\nEvaluating on test set...")
    test_loss, test_preds, test_labels, test_preds_multi, test_labels_multi = (
        val_one_epoch(
            args,
            test_loader,
            model,
            loss_fn,
            loss_audio,
            loss_vision,
            loss_text,
        )
    )
    test_acc, test_f1, test_auc = evaluation(
        test_labels.detach().cpu().numpy(), test_preds.detach().cpu().numpy()
    )
    test_results = "test results: acc {:.5f}, f1: {:.5f}, auc:{:.5f}".format(
        test_acc, test_f1, test_auc
    )
    test_report = classification_report(
        test_labels.cpu().numpy(),
        test_preds.cpu().numpy(),
        target_names=["truth", "deception"],
    )

    print("\nValidation results:\n")
    print(results)
    print("\nTest results:\n")
    print(test_results)
    f.write("****************\n")
    f.write(results)
    f.write("\n\n")
    f.write(val_results)
    f.write("\n\n")
    f.write(test_results)
    f.write("\n\n")
    f.write(test_report)
    f.write("\n\n")
    f.write(multi_task_result)
    
    final_path = save_checkpoint(
        args.checkpoint_dir,
        args.model_name + "_final",
        model,
        optimizer,
        args.num_epochs - 1,
        best_acc,
        model_config,
        is_best=False,
    )
    print(f"\nSaved final model to {final_path}")
    f.write(f"\n\nFinal model saved to: {final_path}")
    f.write(f"\nBest model saved to: {args.checkpoint_dir}/{args.model_name}_best.pt")

    f.close()


if __name__ == "__main__":
    opts = parse_options()

    # For every run create an new entry and register the hyperparameters
    # num_existing_files = len(os.listdir(opts.log))
    log_name = os.path.join(opts.log, str(time.time()) + "_" + opts.model_name + ".txt")
    with open(log_name, "w") as f:
        f.write("\nOptimizer and LR = Adam, " + str(opts.lr))
        f.write("\nBatch Size = " + str(opts.batch_size))
        f.write("\nEpochs = " + str(opts.num_epochs))
        f.write("\nNum Encoders = " + str(opts.num_encoders))
        f.write("\nAdapter = " + str(opts.adapter))
        if opts.adapter:
            f.write("\nAdapter Type = " + opts.adapter_type)
        f.write("\n------------------------------------------")
        f.write("\n------------------------------------------")
    f.close()

    train_test(log_name, args=opts)
